; child-snack task with 2 children and 0.44 gluten factor 
; constant factor of 1.9
; random seed: 209316

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child182 child499 - child
    bread59 bread259 - bread-portion
    content189 content106 - content-portion
    tray237 tray249 tray158 - tray
    table238 table419 table180 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray237 kitchen)
     (at tray249 kitchen)
     (at tray158 kitchen)
     (at_kitchen_bread bread59)
     (at_kitchen_bread bread259)
     (at_kitchen_content content189)
     (at_kitchen_content content106)
     (not_allergic_gluten child499)
     (not_allergic_gluten child182)
     (waiting child182 table238)
     (waiting child499 table238)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child182)
     (served child499)
    )
  )
)
