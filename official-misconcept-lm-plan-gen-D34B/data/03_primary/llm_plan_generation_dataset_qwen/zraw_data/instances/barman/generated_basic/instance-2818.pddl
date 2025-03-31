(define (problem prob)
 (:domain barman)
 (:objects 
      shaker155 - shaker
      left right - hand
      shot467 shot189 shot4 - shot
      ingredient49 ingredient161 ingredient403 ingredient95 - ingredient
      cocktail155 - cocktail
      dispenser168 dispenser311 dispenser238 dispenser34 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker155)
  (ontable shot467)
  (ontable shot189)
  (ontable shot4)
  (dispenses dispenser168 ingredient49)
  (dispenses dispenser311 ingredient161)
  (dispenses dispenser238 ingredient403)
  (dispenses dispenser34 ingredient95)
  (clean shaker155)
  (clean shot467)
  (clean shot189)
  (clean shot4)
  (empty shaker155)
  (empty shot467)
  (empty shot189)
  (empty shot4)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker155 l0)
  (shaker-level shaker155 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail155 ingredient95)
  (cocktail-part2 cocktail155 ingredient403)
)
 (:goal
  (and
      (contains shot467 cocktail155)
      (contains shot189 cocktail155)
)))
