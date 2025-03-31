(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot219 shot115 - shot
      ingredient422 ingredient58 ingredient473 ingredient489 - ingredient
      cocktail97 - cocktail
      dispenser167 dispenser434 dispenser295 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot219)
  (ontable shot115)
  (dispenses dispenser167 ingredient422)
  (dispenses dispenser434 ingredient58)
  (dispenses dispenser295 ingredient473)
  (dispenses dispenser80 ingredient489)
  (clean shaker113)
  (clean shot219)
  (clean shot115)
  (empty shaker113)
  (empty shot219)
  (empty shot115)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail97 ingredient473)
  (cocktail-part2 cocktail97 ingredient422)
)
 (:goal
  (and
      (contains shot219 cocktail97)
)))
