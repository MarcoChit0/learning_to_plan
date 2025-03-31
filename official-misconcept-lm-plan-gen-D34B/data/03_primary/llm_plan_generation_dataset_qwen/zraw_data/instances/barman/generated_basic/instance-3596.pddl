(define (problem prob)
 (:domain barman)
 (:objects 
      shaker278 - shaker
      left right - hand
      shot76 shot388 shot276 shot280 - shot
      ingredient76 ingredient437 ingredient85 - ingredient
      cocktail381 - cocktail
      dispenser304 dispenser44 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker278)
  (ontable shot76)
  (ontable shot388)
  (ontable shot276)
  (ontable shot280)
  (dispenses dispenser304 ingredient76)
  (dispenses dispenser44 ingredient437)
  (dispenses dispenser32 ingredient85)
  (clean shaker278)
  (clean shot76)
  (clean shot388)
  (clean shot276)
  (clean shot280)
  (empty shaker278)
  (empty shot76)
  (empty shot388)
  (empty shot276)
  (empty shot280)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker278 l0)
  (shaker-level shaker278 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail381 ingredient76)
  (cocktail-part2 cocktail381 ingredient437)
)
 (:goal
  (and
      (contains shot76 cocktail381)
      (contains shot388 cocktail381)
      (contains shot276 ingredient76)
)))
