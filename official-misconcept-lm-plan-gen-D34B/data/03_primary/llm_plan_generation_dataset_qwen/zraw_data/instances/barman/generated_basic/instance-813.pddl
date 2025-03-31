(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot270 shot201 - shot
      ingredient449 ingredient490 ingredient395 ingredient203 - ingredient
      cocktail1 - cocktail
      dispenser165 dispenser488 dispenser154 dispenser250 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot270)
  (ontable shot201)
  (dispenses dispenser165 ingredient449)
  (dispenses dispenser488 ingredient490)
  (dispenses dispenser154 ingredient395)
  (dispenses dispenser250 ingredient203)
  (clean shaker360)
  (clean shot270)
  (clean shot201)
  (empty shaker360)
  (empty shot270)
  (empty shot201)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient203)
  (cocktail-part2 cocktail1 ingredient395)
)
 (:goal
  (and
      (contains shot270 cocktail1)
)))
