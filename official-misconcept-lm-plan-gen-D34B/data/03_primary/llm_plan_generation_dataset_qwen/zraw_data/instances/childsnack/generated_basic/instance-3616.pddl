; child-snack task with 2 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 747928

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child39 child65 - child
    bread392 bread396 - bread-portion
    content304 content386 - content-portion
    tray156 tray233 tray216 tray114 - tray
    table206 table221 table429 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray156 kitchen)
     (at tray233 kitchen)
     (at tray216 kitchen)
     (at tray114 kitchen)
     (at_kitchen_bread bread392)
     (at_kitchen_bread bread396)
     (at_kitchen_content content304)
     (at_kitchen_content content386)
     (no_gluten_bread bread392)
     (no_gluten_content content304)
     (allergic_gluten child39)
     (not_allergic_gluten child65)
     (waiting child39 table206)
     (waiting child65 table429)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child39)
     (served child65)
    )
  )
)
