; child-snack task with 2 children and 0.58 gluten factor 
; constant factor of 1.5
; random seed: 45534

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child439 child151 - child
    bread278 bread209 - bread-portion
    content280 content430 - content-portion
    tray194 tray141 tray25 - tray
    table247 table175 table111 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray194 kitchen)
     (at tray141 kitchen)
     (at tray25 kitchen)
     (at_kitchen_bread bread278)
     (at_kitchen_bread bread209)
     (at_kitchen_content content280)
     (at_kitchen_content content430)
     (no_gluten_bread bread278)
     (no_gluten_content content280)
     (allergic_gluten child151)
     (not_allergic_gluten child439)
     (waiting child439 table247)
     (waiting child151 table247)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child439)
     (served child151)
    )
  )
)
