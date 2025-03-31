; child-snack task with 2 children and 0.32 gluten factor 
; constant factor of 1.7
; random seed: 652466

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child109 child336 - child
    bread257 bread450 - bread-portion
    content150 content274 - content-portion
    tray281 tray68 - tray
    table439 table42 table247 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray281 kitchen)
     (at tray68 kitchen)
     (at_kitchen_bread bread257)
     (at_kitchen_bread bread450)
     (at_kitchen_content content150)
     (at_kitchen_content content274)
     (not_allergic_gluten child109)
     (not_allergic_gluten child336)
     (waiting child109 table247)
     (waiting child336 table42)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child109)
     (served child336)
    )
  )
)
