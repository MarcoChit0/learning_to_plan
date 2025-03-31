(define (problem prob)
 (:domain barman)
 (:objects 
      shaker300 - shaker
      left right - hand
      shot295 shot307 shot283 - shot
      ingredient179 ingredient170 ingredient87 ingredient84 - ingredient
      cocktail1 - cocktail
      dispenser99 dispenser56 dispenser467 dispenser180 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker300)
  (ontable shot295)
  (ontable shot307)
  (ontable shot283)
  (dispenses dispenser99 ingredient179)
  (dispenses dispenser56 ingredient170)
  (dispenses dispenser467 ingredient87)
  (dispenses dispenser180 ingredient84)
  (clean shaker300)
  (clean shot295)
  (clean shot307)
  (clean shot283)
  (empty shaker300)
  (empty shot295)
  (empty shot307)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker300 l0)
  (shaker-level shaker300 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient170)
  (cocktail-part2 cocktail1 ingredient84)
)
 (:goal
  (and
      (contains shot295 cocktail1)
      (contains shot307 ingredient179)
)))
