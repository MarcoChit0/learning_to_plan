(define (problem prob)
 (:domain barman)
 (:objects 
      shaker25 - shaker
      left right - hand
      shot122 shot460 shot397 shot346 - shot
      ingredient58 ingredient308 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser482 dispenser313 dispenser96 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker25)
  (ontable shot122)
  (ontable shot460)
  (ontable shot397)
  (ontable shot346)
  (dispenses dispenser482 ingredient58)
  (dispenses dispenser313 ingredient308)
  (dispenses dispenser96 ingredient250)
  (clean shaker25)
  (clean shot122)
  (clean shot460)
  (clean shot397)
  (clean shot346)
  (empty shaker25)
  (empty shot122)
  (empty shot460)
  (empty shot397)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker25 l0)
  (shaker-level shaker25 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient250)
  (cocktail-part2 cocktail1 ingredient58)
)
 (:goal
  (and
      (contains shot122 cocktail1)
      (contains shot460 ingredient250)
      (contains shot397 ingredient308)
)))
