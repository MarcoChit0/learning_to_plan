(define (problem prob)
 (:domain barman)
 (:objects 
      shaker443 - shaker
      left right - hand
      shot177 shot364 shot118 - shot
      ingredient37 ingredient149 ingredient336 ingredient98 - ingredient
      cocktail171 - cocktail
      dispenser420 dispenser332 dispenser465 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker443)
  (ontable shot177)
  (ontable shot364)
  (ontable shot118)
  (dispenses dispenser420 ingredient37)
  (dispenses dispenser332 ingredient149)
  (dispenses dispenser465 ingredient336)
  (dispenses dispenser422 ingredient98)
  (clean shaker443)
  (clean shot177)
  (clean shot364)
  (clean shot118)
  (empty shaker443)
  (empty shot177)
  (empty shot364)
  (empty shot118)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker443 l0)
  (shaker-level shaker443 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient37)
  (cocktail-part2 cocktail171 ingredient149)
)
 (:goal
  (and
      (contains shot177 cocktail171)
      (contains shot364 ingredient336)
)))
