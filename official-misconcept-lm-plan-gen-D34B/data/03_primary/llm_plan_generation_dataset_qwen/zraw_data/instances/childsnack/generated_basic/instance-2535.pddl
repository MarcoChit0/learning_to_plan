; child-snack task with 1 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 216077

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child122 - child
    bread75 - bread-portion
    content376 - content-portion
    tray200 tray448 tray108 - tray
    table154 table45 table49 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray200 kitchen)
     (at tray448 kitchen)
     (at tray108 kitchen)
     (at_kitchen_bread bread75)
     (at_kitchen_content content376)
     (not_allergic_gluten child122)
     (waiting child122 table154)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child122)
    )
  )
)
