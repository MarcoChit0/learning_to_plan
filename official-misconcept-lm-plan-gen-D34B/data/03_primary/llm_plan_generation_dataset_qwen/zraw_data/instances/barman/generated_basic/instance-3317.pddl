(define (problem prob)
 (:domain barman)
 (:objects 
      shaker433 - shaker
      left right - hand
      shot365 shot170 shot32 - shot
      ingredient195 ingredient458 ingredient173 - ingredient
      cocktail368 - cocktail
      dispenser458 dispenser23 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker433)
  (ontable shot365)
  (ontable shot170)
  (ontable shot32)
  (dispenses dispenser458 ingredient195)
  (dispenses dispenser23 ingredient458)
  (dispenses dispenser464 ingredient173)
  (clean shaker433)
  (clean shot365)
  (clean shot170)
  (clean shot32)
  (empty shaker433)
  (empty shot365)
  (empty shot170)
  (empty shot32)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker433 l0)
  (shaker-level shaker433 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail368 ingredient173)
  (cocktail-part2 cocktail368 ingredient195)
)
 (:goal
  (and
      (contains shot365 cocktail368)
      (contains shot170 cocktail368)
)))
