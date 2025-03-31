(define (problem prob)
 (:domain barman)
 (:objects 
      shaker215 - shaker
      left right - hand
      shot332 shot139 shot480 - shot
      ingredient80 ingredient410 ingredient321 ingredient291 - ingredient
      cocktail0 - cocktail
      dispenser368 dispenser372 dispenser41 dispenser165 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker215)
  (ontable shot332)
  (ontable shot139)
  (ontable shot480)
  (dispenses dispenser368 ingredient80)
  (dispenses dispenser372 ingredient410)
  (dispenses dispenser41 ingredient321)
  (dispenses dispenser165 ingredient291)
  (clean shaker215)
  (clean shot332)
  (clean shot139)
  (clean shot480)
  (empty shaker215)
  (empty shot332)
  (empty shot139)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker215 l0)
  (shaker-level shaker215 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail0 ingredient410)
  (cocktail-part2 cocktail0 ingredient321)
)
 (:goal
  (and
      (contains shot332 cocktail0)
      (contains shot139 cocktail0)
)))
