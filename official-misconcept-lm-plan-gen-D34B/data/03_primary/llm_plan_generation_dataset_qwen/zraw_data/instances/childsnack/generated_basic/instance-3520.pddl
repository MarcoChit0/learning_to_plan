; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 444734

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child279 child136 - child
    bread106 bread357 - bread-portion
    content402 content197 - content-portion
    tray495 tray122 - tray
    table422 table144 table382 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray495 kitchen)
     (at tray122 kitchen)
     (at_kitchen_bread bread106)
     (at_kitchen_bread bread357)
     (at_kitchen_content content402)
     (at_kitchen_content content197)
     (not_allergic_gluten child279)
     (not_allergic_gluten child136)
     (waiting child279 table144)
     (waiting child136 table422)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child279)
     (served child136)
    )
  )
)
