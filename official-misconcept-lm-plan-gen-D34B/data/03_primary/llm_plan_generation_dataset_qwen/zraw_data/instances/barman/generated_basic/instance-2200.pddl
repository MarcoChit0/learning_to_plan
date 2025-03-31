(define (problem prob)
 (:domain barman)
 (:objects 
      shaker251 - shaker
      left right - hand
      shot324 shot420 shot75 - shot
      ingredient247 ingredient30 ingredient41 ingredient289 - ingredient
      cocktail408 - cocktail
      dispenser111 dispenser201 dispenser327 dispenser407 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker251)
  (ontable shot324)
  (ontable shot420)
  (ontable shot75)
  (dispenses dispenser111 ingredient247)
  (dispenses dispenser201 ingredient30)
  (dispenses dispenser327 ingredient41)
  (dispenses dispenser407 ingredient289)
  (clean shaker251)
  (clean shot324)
  (clean shot420)
  (clean shot75)
  (empty shaker251)
  (empty shot324)
  (empty shot420)
  (empty shot75)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker251 l0)
  (shaker-level shaker251 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail408 ingredient41)
  (cocktail-part2 cocktail408 ingredient289)
)
 (:goal
  (and
      (contains shot324 cocktail408)
      (contains shot420 ingredient30)
)))
