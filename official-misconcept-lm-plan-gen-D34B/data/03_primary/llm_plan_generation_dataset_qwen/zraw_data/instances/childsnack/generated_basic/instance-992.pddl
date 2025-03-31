; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 84512

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child115 child258 child432 - child
    bread93 bread477 bread221 - bread-portion
    content110 content144 content312 - content-portion
    tray371 - tray
    table162 table466 table479 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray371 kitchen)
     (at_kitchen_bread bread93)
     (at_kitchen_bread bread477)
     (at_kitchen_bread bread221)
     (at_kitchen_content content110)
     (at_kitchen_content content144)
     (at_kitchen_content content312)
     (not_allergic_gluten child432)
     (not_allergic_gluten child115)
     (not_allergic_gluten child258)
     (waiting child115 table466)
     (waiting child258 table162)
     (waiting child432 table479)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child115)
     (served child258)
     (served child432)
    )
  )
)
