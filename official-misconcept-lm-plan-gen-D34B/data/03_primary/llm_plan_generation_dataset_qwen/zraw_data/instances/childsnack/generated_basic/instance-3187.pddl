; child-snack task with 2 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 577299

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child364 child205 - child
    bread396 bread211 - bread-portion
    content359 content484 - content-portion
    tray175 tray52 - tray
    table317 table477 table57 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at tray52 kitchen)
     (at_kitchen_bread bread396)
     (at_kitchen_bread bread211)
     (at_kitchen_content content359)
     (at_kitchen_content content484)
     (no_gluten_bread bread211)
     (no_gluten_content content359)
     (allergic_gluten child205)
     (not_allergic_gluten child364)
     (waiting child364 table57)
     (waiting child205 table477)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child364)
     (served child205)
    )
  )
)
