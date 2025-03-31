(define (problem prob)
 (:domain barman)
 (:objects 
      shaker151 - shaker
      left right - hand
      shot238 shot75 - shot
      ingredient174 ingredient97 ingredient328 ingredient209 - ingredient
      cocktail91 - cocktail
      dispenser90 dispenser60 dispenser390 dispenser230 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker151)
  (ontable shot238)
  (ontable shot75)
  (dispenses dispenser90 ingredient174)
  (dispenses dispenser60 ingredient97)
  (dispenses dispenser390 ingredient328)
  (dispenses dispenser230 ingredient209)
  (clean shaker151)
  (clean shot238)
  (clean shot75)
  (empty shaker151)
  (empty shot238)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker151 l0)
  (shaker-level shaker151 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail91 ingredient328)
  (cocktail-part2 cocktail91 ingredient174)
)
 (:goal
  (and
      (contains shot238 cocktail91)
)))
