; child-snack task with 2 children and 0.49 gluten factor 
; constant factor of 1.7
; random seed: 978865

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child206 child205 - child
    bread465 bread72 - bread-portion
    content291 content396 - content-portion
    tray320 tray240 tray62 tray266 - tray
    table489 table143 table422 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray320 kitchen)
     (at tray240 kitchen)
     (at tray62 kitchen)
     (at tray266 kitchen)
     (at_kitchen_bread bread465)
     (at_kitchen_bread bread72)
     (at_kitchen_content content291)
     (at_kitchen_content content396)
     (not_allergic_gluten child206)
     (not_allergic_gluten child205)
     (waiting child206 table143)
     (waiting child205 table143)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child206)
     (served child205)
    )
  )
)
