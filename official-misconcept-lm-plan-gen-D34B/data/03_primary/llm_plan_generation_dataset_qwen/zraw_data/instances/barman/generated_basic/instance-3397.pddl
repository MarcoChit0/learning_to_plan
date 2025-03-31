(define (problem prob)
 (:domain barman)
 (:objects 
      shaker157 - shaker
      left right - hand
      shot224 shot319 - shot
      ingredient154 ingredient359 ingredient75 ingredient192 - ingredient
      cocktail436 - cocktail
      dispenser376 dispenser453 dispenser264 dispenser107 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker157)
  (ontable shot224)
  (ontable shot319)
  (dispenses dispenser376 ingredient154)
  (dispenses dispenser453 ingredient359)
  (dispenses dispenser264 ingredient75)
  (dispenses dispenser107 ingredient192)
  (clean shaker157)
  (clean shot224)
  (clean shot319)
  (empty shaker157)
  (empty shot224)
  (empty shot319)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker157 l0)
  (shaker-level shaker157 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail436 ingredient75)
  (cocktail-part2 cocktail436 ingredient359)
)
 (:goal
  (and
      (contains shot224 cocktail436)
)))
