(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot41 shot60 - shot
      ingredient263 ingredient337 ingredient162 ingredient15 - ingredient
      cocktail170 - cocktail
      dispenser254 dispenser344 dispenser355 dispenser356 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot41)
  (ontable shot60)
  (dispenses dispenser254 ingredient263)
  (dispenses dispenser344 ingredient337)
  (dispenses dispenser355 ingredient162)
  (dispenses dispenser356 ingredient15)
  (clean shaker183)
  (clean shot41)
  (clean shot60)
  (empty shaker183)
  (empty shot41)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail170 ingredient337)
  (cocktail-part2 cocktail170 ingredient15)
)
 (:goal
  (and
      (contains shot41 cocktail170)
)))
