(define (problem prob)
 (:domain barman)
 (:objects 
      shaker143 - shaker
      left right - hand
      shot301 shot20 shot96 - shot
      ingredient226 ingredient209 - ingredient
      cocktail247 - cocktail
      dispenser483 dispenser32 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker143)
  (ontable shot301)
  (ontable shot20)
  (ontable shot96)
  (dispenses dispenser483 ingredient226)
  (dispenses dispenser32 ingredient209)
  (clean shaker143)
  (clean shot301)
  (clean shot20)
  (clean shot96)
  (empty shaker143)
  (empty shot301)
  (empty shot20)
  (empty shot96)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker143 l0)
  (shaker-level shaker143 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail247 ingredient209)
  (cocktail-part2 cocktail247 ingredient226)
)
 (:goal
  (and
      (contains shot301 cocktail247)
      (contains shot20 ingredient226)
)))
