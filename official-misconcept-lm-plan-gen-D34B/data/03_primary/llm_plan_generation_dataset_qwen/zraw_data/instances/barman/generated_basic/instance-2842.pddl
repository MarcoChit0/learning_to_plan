(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot384 shot146 - shot
      ingredient173 ingredient175 ingredient338 ingredient449 - ingredient
      cocktail104 - cocktail
      dispenser247 dispenser298 dispenser212 dispenser255 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot384)
  (ontable shot146)
  (dispenses dispenser247 ingredient173)
  (dispenses dispenser298 ingredient175)
  (dispenses dispenser212 ingredient338)
  (dispenses dispenser255 ingredient449)
  (clean shaker292)
  (clean shot384)
  (clean shot146)
  (empty shaker292)
  (empty shot384)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail104 ingredient175)
  (cocktail-part2 cocktail104 ingredient173)
)
 (:goal
  (and
      (contains shot384 cocktail104)
)))
