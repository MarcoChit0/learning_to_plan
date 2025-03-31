(define (problem prob)
 (:domain barman)
 (:objects 
      shaker49 - shaker
      left right - hand
      shot42 shot269 shot155 - shot
      ingredient165 ingredient1 ingredient126 ingredient455 - ingredient
      cocktail264 - cocktail
      dispenser209 dispenser189 dispenser258 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker49)
  (ontable shot42)
  (ontable shot269)
  (ontable shot155)
  (dispenses dispenser209 ingredient165)
  (dispenses dispenser189 ingredient1)
  (dispenses dispenser258 ingredient126)
  (dispenses dispenser240 ingredient455)
  (clean shaker49)
  (clean shot42)
  (clean shot269)
  (clean shot155)
  (empty shaker49)
  (empty shot42)
  (empty shot269)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker49 l0)
  (shaker-level shaker49 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail264 ingredient1)
  (cocktail-part2 cocktail264 ingredient165)
)
 (:goal
  (and
      (contains shot42 cocktail264)
      (contains shot269 cocktail264)
)))
