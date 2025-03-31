; child-snack task with 2 children and 0.55 gluten factor 
; constant factor of 1.7
; random seed: 492433

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child488 child380 - child
    bread37 bread75 - bread-portion
    content484 content441 - content-portion
    tray83 tray176 tray21 tray308 - tray
    table287 table235 table168 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray83 kitchen)
     (at tray176 kitchen)
     (at tray21 kitchen)
     (at tray308 kitchen)
     (at_kitchen_bread bread37)
     (at_kitchen_bread bread75)
     (at_kitchen_content content484)
     (at_kitchen_content content441)
     (no_gluten_bread bread75)
     (no_gluten_content content441)
     (allergic_gluten child380)
     (not_allergic_gluten child488)
     (waiting child488 table235)
     (waiting child380 table168)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child488)
     (served child380)
    )
  )
)
