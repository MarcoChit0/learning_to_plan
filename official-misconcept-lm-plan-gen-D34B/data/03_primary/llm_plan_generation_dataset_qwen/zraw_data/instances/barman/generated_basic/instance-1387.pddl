(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot50 shot378 shot251 - shot
      ingredient236 ingredient282 ingredient351 ingredient168 - ingredient
      cocktail423 - cocktail
      dispenser311 dispenser44 dispenser60 dispenser134 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot50)
  (ontable shot378)
  (ontable shot251)
  (dispenses dispenser311 ingredient236)
  (dispenses dispenser44 ingredient282)
  (dispenses dispenser60 ingredient351)
  (dispenses dispenser134 ingredient168)
  (clean shaker321)
  (clean shot50)
  (clean shot378)
  (clean shot251)
  (empty shaker321)
  (empty shot50)
  (empty shot378)
  (empty shot251)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail423 ingredient351)
  (cocktail-part2 cocktail423 ingredient282)
)
 (:goal
  (and
      (contains shot50 cocktail423)
      (contains shot378 ingredient168)
)))
