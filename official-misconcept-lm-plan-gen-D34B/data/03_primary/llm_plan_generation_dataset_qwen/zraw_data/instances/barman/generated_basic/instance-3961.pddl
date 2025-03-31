(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot78 shot456 - shot
      ingredient65 ingredient186 ingredient212 ingredient85 - ingredient
      cocktail260 - cocktail
      dispenser274 dispenser328 dispenser342 dispenser336 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot78)
  (ontable shot456)
  (dispenses dispenser274 ingredient65)
  (dispenses dispenser328 ingredient186)
  (dispenses dispenser342 ingredient212)
  (dispenses dispenser336 ingredient85)
  (clean shaker64)
  (clean shot78)
  (clean shot456)
  (empty shaker64)
  (empty shot78)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient212)
  (cocktail-part2 cocktail260 ingredient85)
)
 (:goal
  (and
      (contains shot78 cocktail260)
)))
