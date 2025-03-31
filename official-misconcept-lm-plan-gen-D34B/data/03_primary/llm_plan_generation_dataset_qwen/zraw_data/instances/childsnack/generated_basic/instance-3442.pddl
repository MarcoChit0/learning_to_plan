; child-snack task with 1 children and 0.77 gluten factor 
; constant factor of 1.9
; random seed: 201240

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child382 - child
    bread496 - bread-portion
    content329 - content-portion
    tray382 tray330 tray259 tray45 - tray
    table137 table479 table115 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray382 kitchen)
     (at tray330 kitchen)
     (at tray259 kitchen)
     (at tray45 kitchen)
     (at_kitchen_bread bread496)
     (at_kitchen_content content329)
     (not_allergic_gluten child382)
     (waiting child382 table115)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child382)
    )
  )
)
