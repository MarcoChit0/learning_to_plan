(define (problem prob)
 (:domain barman)
 (:objects 
      shaker305 - shaker
      left right - hand
      shot105 shot413 shot251 shot452 - shot
      ingredient169 ingredient232 ingredient6 ingredient438 - ingredient
      cocktail260 - cocktail
      dispenser405 dispenser14 dispenser154 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker305)
  (ontable shot105)
  (ontable shot413)
  (ontable shot251)
  (ontable shot452)
  (dispenses dispenser405 ingredient169)
  (dispenses dispenser14 ingredient232)
  (dispenses dispenser154 ingredient6)
  (dispenses dispenser33 ingredient438)
  (clean shaker305)
  (clean shot105)
  (clean shot413)
  (clean shot251)
  (clean shot452)
  (empty shaker305)
  (empty shot105)
  (empty shot413)
  (empty shot251)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker305 l0)
  (shaker-level shaker305 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail260 ingredient232)
  (cocktail-part2 cocktail260 ingredient6)
)
 (:goal
  (and
      (contains shot105 cocktail260)
      (contains shot413 cocktail260)
      (contains shot251 ingredient6)
)))
