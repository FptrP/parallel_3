#include <mpi.h>
#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>
#include <omp.h>

struct TimeProfiler
{
  TimeProfiler(double &out)
  {
    measuredTime = &out;
    t_start = omp_get_wtime();
  }

  void stop()
  {
    if (!measuredTime)
      return;
    double t_end = omp_get_wtime();
    *measuredTime += 1000.0 * (t_end - t_start); 
    measuredTime = NULL;
  }

  ~TimeProfiler()
  {
    stop();
  }

  double *measuredTime;
  double t_start;
};

struct SolveParams
{
  // dimensions
  double Lx;
  double Ly;
  double Lz;
  //time
  double T;

  //a^2
  double a2;

  int num_threads;
  int dim_steps; // per Lx Ly Lz
  int time_steps; 

  int num_blocks;
  int block_id;

  int num_block_x;
  int num_block_y;
  int num_block_z;

  int block_dim_x;
  int block_dim_y;
  int block_dim_z;

  int block_x;
  int block_y;
  int block_z;

  void init()
  {
    if (!(Lx == Ly && Ly == Lz))
      throw "Invalid grid size";

    h = Lx/(dim_steps - 1);
    tau = T/(time_steps - 1);
  
    MPI_Comm_size(MPI_COMM_WORLD, &num_blocks);
    MPI_Comm_rank(MPI_COMM_WORLD, &block_id);
    
    // init dims. Assume that num_blocks is 2^n
    num_block_x = 1;
    num_block_y = 1;
    num_block_z = 1;
    
    int n = num_blocks;
    int power = 0;
    
    while (n > 1)
    {
      power++;
      assert(n % 2 == 0);
      n /= 2;
    }

    int p = power / 3; 
    int k = power % 3; // 0 1 2
    
    if (p)
    {
      num_block_x *= 1 << p;
      num_block_y *= 1 << p;
      num_block_z *= 1 << p;
    }

    if (k > 0)
    {
      num_block_x *= 2;
      k--;
    }

    if (k > 0)
    {
      num_block_y *= 2;
      k--;
    }

    block_dim_x = dim_steps/num_block_x;
    block_dim_y = dim_steps/num_block_y;
    block_dim_z = dim_steps/num_block_z;

    block_x = block_id % num_block_x;
    block_y = (block_id/num_block_x) % num_block_y;
    block_z = block_id/(num_block_x * num_block_y);
    
    if (block_id != 0)
      return;

    std::cout << "Inited process " << block_id << " Dims " 
      << num_block_x << " " << num_block_y << " " << num_block_z 
      << " Addr : " << block_x << " " <<  block_y << " " << block_z
      << " BlkSize : " << block_dim_x << " " << block_dim_y << " " << block_dim_z
      << std::endl;
  }

  int get_rank(int x, int y, int z) const
  {
    assert(0 <= x && x < num_block_x && 0 <= y && y <= num_block_y && 0 <= z && z <= num_block_z);
    return x + (y + z * num_block_y) * num_block_x;
  }

  double h;
  double tau;
};

struct PointGrid
{
  PointGrid() : dimX(0), dimY(0), dimZ(0) {}

  PointGrid(int dx, int dy, int dz)
  {
    init(dx, dy, dz);
  }

  void init(int dx, int dy, int dz)
  {
    dimX = dx; dimY = dy; dimZ = dz;
    points.resize(dimX * dimY * dimZ, 0.0);
  }

  double &operator()(int x, int y, int z)
  {
    return points.at(z * (dimX * dimY) + y * dimX + x);
  }

  double operator()(int x, int y, int z) const
  {
    return points.at(z * (dimX * dimY) + y * dimX + x);
  }

  void fill(double val)
  {
    for (int i = 0; i < dimX * dimY * dimZ; i++)
      points[i] = val;
  }

  int dimX, dimY, dimZ;
  std::vector<double> points;
};

struct MpiBuffers
{


};

static inline double absd(double v)
{
  return (v > 0.0) ? v : (-v);
}

double grid_error(const SolveParams &p, const PointGrid &g1, const PointGrid &g2)
{
  double error = 0.0;

  for (int zs = 1; zs <= p.block_dim_z; zs++)
  {
    for (int ys = 1; ys <= p.block_dim_y; ys++)
    {
      for (int xs = 1; xs <= p.block_dim_x; xs++)
      {
        double v = (g1(xs, ys, zs) - g2(xs, ys, zs));
        v = (v < 0.0)? (-v) : v; 
        error = (v > error) ? v : error;
      }
    }
  }

  return error;
}

//yaml
void save_grid(const SolveParams &p, const PointGrid &g, int time_step, const char *grid_name)
{
  std::stringstream file_name;
  file_name << grid_name << "_" << time_step << ".yaml";
  
  std::string out_name = file_name.str();
  std::ofstream out(out_name.c_str());
  
  out << "name : " << grid_name << std::endl;
  out << "L : " << p.Lx << std::endl;
  out << "T : " << p.T << std::endl;
  out << "time_steps : " << p.time_steps << std::endl;
  out << "time_step : " << time_step << std::endl; 
  out << "dim : " << g.dimX << std::endl;
  out << "points:" << std::endl;

  for (int i = 0; i < g.points.size(); i++)
    out << "- " << g.points[i] << std::endl;
  out.close();
}

// variant 6, x = П y = 1Р z = П
// x: u(0, y, z, t) = u(Lx, y, z, t); dudx(0, y, z, t) = dudx(Lx, y, z, t)
// y: u(x, 0, z, t) = 0; u(x, Ly, z, t) = 0
// z: u(x, y, 0, t) = u(x, y, Lz, t); dudz(x, y, 0, t) = dudz(x, y, Lz, t)

typedef double (*TargetFunction)(const SolveParams &p, double x, double y, double z, double t);

double target_function(const SolveParams &p, double x, double y, double z, double t)
{
  double v1 = sin(2.0 * M_PI/p.Lx * x);
  double v2 = sin(M_PI/p.Ly * y + M_PI);
  double v3 = sin(2.0 * M_PI/p.Lz * z + 2.0 * M_PI);

  const double at = (M_PI/3.0) * sqrt(4.0/(p.Lx * p.Lx) + 1.0/(p.Ly * p.Ly) + 4.0/(p.Lz * p.Lz));
  double v4 = cos(at * t + M_PI);

  return  v1 * v2 * v3 * v4;
}

double laplacian(const SolveParams &p, const PointGrid &g, int x, int y, int z)
{
  double res = 0.0;
  double h2 = p.h * p.h;
  double uc = g(x, y, z);

  int x1 = (x - 1);
  int x2 = x + 1;

  int z1 = (z - 1);
  int z2 = z + 1;

  res += g(x1, y, z) - 2.0 * uc + g(x2, y, z);
  res += g(x, y - 1, z) - 2.0 * uc + g(x, y + 1, z);
  res += g(x, y, z1) - 2.0 * uc + g(x, y, z2);

  return res/h2; 
}

void fill_grid(const SolveParams &p, PointGrid &g, double t)
{
  #pragma omp parallel for
  for (int zs = 1; zs <= p.block_dim_z; zs++)
  {
    for (int ys = 1; ys <= p.block_dim_y; ys++)
    {
      for (int xs = 1; xs <= p.block_dim_x; xs++)
      {
        double x = (xs - 1 + p.block_x * p.block_dim_x) * p.h;
        double y = (ys - 1 + p.block_y * p.block_dim_y) * p.h;
        double z = (zs - 1 + p.block_z * p.block_dim_z) * p.h;
        g(xs, ys, zs) = target_function(p, x, y, z, t);
      }
    }
  }
}

enum GridSide
{
  XP,
  XN,
  YP,
  YN,
  ZP,
  ZN
};

int elems_count(const SolveParams &p, GridSide side)
{
  switch (side)
  {
    case XP:
    case XN:
      return p.block_dim_y * p.block_dim_z;
      break;
    case YP:
    case YN:
      return p.block_dim_x * p.block_dim_z;
      break;
    case ZP:
    case ZN:
      return p.block_dim_x * p.block_dim_y;
      break;
    default: assert(0);
  }
  return 0;
}

GridSide inverse_side(GridSide side)
{
  switch (side)
  {
    case XP: return XN;
    case XN: return XP;
    case YP: return YN;
    case YN: return YP;
    case ZP: return ZN;
    case ZN: return ZP;
    default: assert(0);
  }
  return XP;
}

int get_neighbour(const SolveParams &p, GridSide side)
{
  switch (side)
  {
    case XP:
      return p.get_rank((p.block_x + 1) % p.num_block_x, p.block_y, p.block_z);
    case XN:
      return p.get_rank((p.block_x - 1 + p.num_block_x) % p.num_block_x, p.block_y, p.block_z);
    case YP:
      return p.get_rank(p.block_x, (p.block_y + 1) % p.num_block_y, p.block_z);
    case YN:
      return p.get_rank(p.block_x, (p.block_y - 1 + p.num_block_y) % p.num_block_y, p.block_z);
    case ZP:
      return p.get_rank(p.block_x, p.block_y, (p.block_z + 1) % p.num_block_z);
    case ZN:
      return p.get_rank(p.block_x, p.block_y, (p.block_z - 1 + p.num_block_z) % p.num_block_z);
    default: assert(0);
  }
  return -1;
}

void send_data(const SolveParams &p, PointGrid &g, std::vector<double> &buffer, GridSide side)
{
  int send_dst = get_neighbour(p, side);
  int count = elems_count(p, side);
  int coord = -1;
  switch (side)
  {
    case XP:
      coord = (p.block_x != p.num_block_x - 1)? p.block_dim_x : (p.block_dim_x - 1);
      for (int y = 1; y <= p.block_dim_y; y++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          buffer[(y - 1) + (z - 1) * p.block_dim_y] = g(coord, y, z);
        }
      }
      break;
    case XN:
      coord = (p.block_x != 0)? 1 : 2;
      for (int y = 1; y <= p.block_dim_y; y++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          buffer[(y - 1) + (z - 1) * p.block_dim_y] = g(coord, y, z);
        }
      }
      break;
    case YP:
      coord = (p.block_y != p.num_block_y - 1)? p.block_dim_y : (p.block_dim_y - 1);
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          buffer[(x - 1) + (z - 1) * p.block_dim_x] = g(x, coord, z);
        }
      }
      break;
    case YN:
      coord = (p.block_y != 0)? 1 : 2;
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          buffer[(x - 1) + (z - 1) * p.block_dim_x] = g(x, coord, z);
        }
      }
      break;
    case ZP:
      coord = (p.block_z != p.num_block_z - 1)? p.block_dim_z : (p.block_dim_z - 1);
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int y = 1; y <= p.block_dim_y; y++)
        {
          buffer[(x - 1) + (y - 1) * p.block_dim_x] = g(x, y, coord);
        }
      }
      break;
    case ZN:
      coord = (p.block_z != 0)? 1 : 2;
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int y = 1; y <= p.block_dim_y; y++)
        {
          buffer[(x - 1) + (y - 1) * p.block_dim_x] = g(x, y, coord);
        }
      }
      break;
    default: assert(0);
  }

  //std::cout << "Send from " << p.block_id << " to " << send_dst << " " << count << "\n";
  MPI_Send(buffer.data(), count, MPI_DOUBLE, send_dst, 0, MPI_COMM_WORLD);
}

void recv_data(const SolveParams &p, PointGrid &g, std::vector<double> &buffer, GridSide side)
{
  int sender_src = get_neighbour(p, inverse_side(side));
  int count = elems_count(p, side);
  MPI_Status status;
  //std::cout << "Recv from " << sender_src << " to " << p.block_id << " " << count << "\n";
  MPI_Recv(buffer.data(), count, MPI_DOUBLE, sender_src, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

  int coord = -1;

  switch (side)
  {
    case XP:
      coord = 0;
      for (int y = 1; y <= p.block_dim_y; y++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          g(coord, y, z) = buffer[(y - 1) + (z - 1) * p.block_dim_y];
        }
      }
      break;
    case XN:
      coord = p.block_dim_x + 1;
      for (int y = 1; y <= p.block_dim_y; y++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          g(coord, y, z) = buffer[(y - 1) + (z - 1) * p.block_dim_y];
        }
      }
      break;
    case YP:
      coord = 0;
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          g(x, coord, z) = buffer[(x - 1) + (z - 1) * p.block_dim_x];
        }
      }
      break;
    case YN:
      coord = p.block_dim_y + 1;
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int z = 1; z <= p.block_dim_z; z++)
        {
          g(x, coord, z) = buffer[(x - 1) + (z - 1) * p.block_dim_x];
        }
      }
      break;
    case ZP:
      coord = 0;
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int y = 1; y <= p.block_dim_y; y++)
        {
          g(x, y, coord) = buffer[(x - 1) + (y - 1) * p.block_dim_x];
        }
      }
      break;
    case ZN:
      coord = p.block_dim_z + 1;
      for (int x = 1; x <= p.block_dim_x; x++)
      {
        for (int y = 1; y <= p.block_dim_y; y++)
        {
          g(x, y, coord) = buffer[(x - 1) + (y - 1) * p.block_dim_x];
        }
      }
      break;
    default: assert(0);
  }
}

void exchange_data(const SolveParams &p, PointGrid &g, std::vector<double> &buffer)
{
  //blk_x
  // (blk_x % 2) == 0 sends first
  //
  if (p.num_block_x == 1)
  {
    for (int y = 1; y <= p.block_dim_y; y++)
    {
      for (int z = 1; z <= p.block_dim_z; z++)
      {
        //
        g(0, y, z) = g(p.block_dim_x - 1, y, z);
        g(p.block_dim_x + 1, y, z) = g(2, y, z);
      }
    }
  }
  else
  {
    if (p.block_x % 2 == 0)
    {
      send_data(p, g, buffer, XP);
      recv_data(p, g, buffer, XP);
      send_data(p, g, buffer, XN);
      recv_data(p, g, buffer, XN);
    }
    else
    {
      recv_data(p, g, buffer, XP);
      send_data(p, g, buffer, XP);
      recv_data(p, g, buffer, XN);
      send_data(p, g, buffer, XN);
    }
  }

  if (p.num_block_y > 1) // if 1 don't exchange at all
  {
    if (p.block_y % 2 == 0)
    {
      send_data(p, g, buffer, YP);
      recv_data(p, g, buffer, YP);
      send_data(p, g, buffer, YN);
      recv_data(p, g, buffer, YN);
    }
    else
    {
      recv_data(p, g, buffer, YP);
      send_data(p, g, buffer, YP);
      recv_data(p, g, buffer, YN);
      send_data(p, g, buffer, YN);
    }
  }

  if (p.num_block_z == 1)
  {
    for (int x = 1; x <= p.block_dim_x; x++)
    {
      for (int y = 1; y <= p.block_dim_y; y++)
      {
        g(x, y, 0) = g(x, y, p.block_dim_z - 1);
        g(x, y, p.block_dim_z + 1) = g(x, y, 2);
      }
    }
  }
  else
  {
    if (p.block_z % 2 == 0)
    {
      send_data(p, g, buffer, ZP);
      recv_data(p, g, buffer, ZP);
      send_data(p, g, buffer, ZN);
      recv_data(p, g, buffer, ZN);
    }
    else
    {
      recv_data(p, g, buffer, ZP);
      send_data(p, g, buffer, ZP);
      recv_data(p, g, buffer, ZN);
      send_data(p, g, buffer, ZN);
    }
  }
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  // argv[1] L argv[2] dim
  
  
  SolveParams params;
  params.Lx = params.Ly = params.Lz = atof(argv[1]);
  params.a2 = 1.0/9.0; // variant 6
  params.dim_steps = atoi(argv[2]);
  params.num_threads = omp_get_max_threads();
  params.T = params.Lx/(params.dim_steps - 1) * 5.0; // h/10 * time_steps
  params.time_steps = 50;

  params.init();
  if (params.block_id == 0)
  {
    std::cout << "T = " << params.T << "\n";
    std::cout << "Tau = " << params.tau << "\n";
  }
  
  std::vector<PointGrid> grids(3);
  for (int i = 0; i < 3; i++)
    grids[i].init(params.block_dim_x + 2, params.block_dim_y + 2, params.block_dim_z + 2);

  PointGrid referenceGrid;
  referenceGrid.init(params.block_dim_x + 2, params.block_dim_y + 2, params.block_dim_z + 2);

  double time = 0.0;
  TimeProfiler init_timer(time);
  fill_grid(params, grids[0], 0.0);

  int max_dim = std::max(params.block_dim_x, std::max(params.block_dim_y, params.block_dim_z));

  std::vector<double> mpi_buffer(max_dim * max_dim);

  exchange_data(params, grids[0], mpi_buffer);

  int ys = (params.block_y != 0)? 1 : 2;
  int ye = (params.block_y != params.num_block_y - 1)? params.block_dim_y : (params.block_dim_y - 1);
  
  #pragma omp parallel for
  for (int z = 1; z <= params.block_dim_z; z++)
  {
    for (int y = ys; y <= ye; y++)
    {
      for (int x = 1; x <= params.block_dim_x; x++)
      {
        grids[1](x, y, z) = grids[0](x, y, z) 
          + 0.5 * params.a2 * params.tau * params.tau * laplacian(params, grids[0], x, y, z); 
      }
    }
  }

  fill_grid(params, referenceGrid, params.tau);

  init_timer.stop();
  double err = grid_error(params, grids[1], referenceGrid);

  for (int ts = 2; ts < params.time_steps; ts++)
  {
    TimeProfiler iter_counter(time);

    PointGrid &g = grids[ts % 3];
    PointGrid &g1 = grids[(ts - 1) % 3];
    PointGrid &g2 = grids[(ts - 2) % 3];

    //std::cout << "RK " << params.block_id << " running iter = " << ts << std::endl;

    exchange_data(params, g1, mpi_buffer);
    #pragma omp parallel for
    for (int z = 1; z <= params.block_dim_z; z++)
    {
      for (int y = ys; y <= ye; y++)
      {
        for (int x = 1; x <= params.block_dim_x; x++)
        {
          double un = g1(x, y, z);
          double un_1 = g2(x, y, z);
          double l_un = laplacian(params, g1, x, y, z);
          g(x, y, z) = params.tau * params.tau * (params.a2 * l_un) + 2.0 * un - un_1;
        }
      }
    }
    
    iter_counter.stop();

    fill_grid(params, referenceGrid, ts * params.tau);
    err = std::max(err, grid_error(params, referenceGrid, g));
    //std::cout << "Grid error " << err << std::endl;  
  }

  //std::cout << "Grid error " << err << std::endl;
  double max_time = 0.0;
  double max_error = 0.0;
  MPI_Reduce(&time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&err, &max_error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (params.block_id == 0)
  {
    std::cout << "Finished " << std::endl;
    std::cout << "Num processes " << params.num_blocks << std::endl;
    std::cout << "Num threads " << omp_get_max_threads() << std::endl;
    std::cout << "Time(s) " << max_time/1000.0 << std::endl;
    std::cout << "Max error " << max_error << std::endl;
  }

  MPI_Finalize();
  return 0;
}