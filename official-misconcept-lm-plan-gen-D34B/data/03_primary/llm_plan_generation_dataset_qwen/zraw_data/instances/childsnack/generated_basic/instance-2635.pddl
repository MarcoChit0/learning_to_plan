; child-snack task with 1 children and 0.35 gluten factor 
; constant factor of 1.3
; random seed: 18041

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child141 - child
    bread150 - bread-portion
    content375 - content-portion
    tray97 tray120 tray186 tray221 - tray
    table414 table110 table49 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray97 kitchen)
     (at tray120 kitchen)
     (at tray186 kitchen)
     (at tray221 kitchen)
     (at_kitchen_bread bread150)
     (at_kitchen_content content375)
     (not_allergic_gluten child141)
     (waiting child141 table110)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child141)
    )
  )
)
