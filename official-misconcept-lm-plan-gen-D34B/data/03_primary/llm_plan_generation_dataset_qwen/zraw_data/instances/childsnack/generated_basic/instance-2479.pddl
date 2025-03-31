; child-snack task with 2 children and 0.39 gluten factor 
; constant factor of 1.9
; random seed: 915940

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child144 child385 - child
    bread229 bread195 - bread-portion
    content262 content305 - content-portion
    tray203 tray453 - tray
    table58 table492 table248 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray203 kitchen)
     (at tray453 kitchen)
     (at_kitchen_bread bread229)
     (at_kitchen_bread bread195)
     (at_kitchen_content content262)
     (at_kitchen_content content305)
     (not_allergic_gluten child385)
     (not_allergic_gluten child144)
     (waiting child144 table58)
     (waiting child385 table58)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child144)
     (served child385)
    )
  )
)
