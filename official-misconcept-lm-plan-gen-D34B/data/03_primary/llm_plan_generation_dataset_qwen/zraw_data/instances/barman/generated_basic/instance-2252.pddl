(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot283 shot421 shot89 - shot
      ingredient469 ingredient495 ingredient326 ingredient238 - ingredient
      cocktail448 - cocktail
      dispenser180 dispenser472 dispenser206 dispenser355 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot283)
  (ontable shot421)
  (ontable shot89)
  (dispenses dispenser180 ingredient469)
  (dispenses dispenser472 ingredient495)
  (dispenses dispenser206 ingredient326)
  (dispenses dispenser355 ingredient238)
  (clean shaker1)
  (clean shot283)
  (clean shot421)
  (clean shot89)
  (empty shaker1)
  (empty shot283)
  (empty shot421)
  (empty shot89)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail448 ingredient495)
  (cocktail-part2 cocktail448 ingredient469)
)
 (:goal
  (and
      (contains shot283 cocktail448)
      (contains shot421 cocktail448)
)))
