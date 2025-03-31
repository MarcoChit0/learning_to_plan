(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot414 shot81 shot22 - shot
      ingredient353 ingredient175 ingredient111 ingredient340 - ingredient
      cocktail1 - cocktail
      dispenser269 dispenser82 dispenser49 dispenser203 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot414)
  (ontable shot81)
  (ontable shot22)
  (dispenses dispenser269 ingredient353)
  (dispenses dispenser82 ingredient175)
  (dispenses dispenser49 ingredient111)
  (dispenses dispenser203 ingredient340)
  (clean shaker469)
  (clean shot414)
  (clean shot81)
  (clean shot22)
  (empty shaker469)
  (empty shot414)
  (empty shot81)
  (empty shot22)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient175)
  (cocktail-part2 cocktail1 ingredient111)
)
 (:goal
  (and
      (contains shot414 cocktail1)
      (contains shot81 cocktail1)
)))
