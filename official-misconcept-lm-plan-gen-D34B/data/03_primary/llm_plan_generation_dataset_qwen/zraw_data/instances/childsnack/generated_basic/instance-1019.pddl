; child-snack task with 3 children and 0.3 gluten factor 
; constant factor of 1.3
; random seed: 902433

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child224 child18 child446 - child
    bread7 bread257 bread88 - bread-portion
    content235 content330 content201 - content-portion
    tray450 - tray
    table213 table48 table71 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray450 kitchen)
     (at_kitchen_bread bread7)
     (at_kitchen_bread bread257)
     (at_kitchen_bread bread88)
     (at_kitchen_content content235)
     (at_kitchen_content content330)
     (at_kitchen_content content201)
     (not_allergic_gluten child446)
     (not_allergic_gluten child18)
     (not_allergic_gluten child224)
     (waiting child224 table48)
     (waiting child18 table213)
     (waiting child446 table71)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child224)
     (served child18)
     (served child446)
    )
  )
)
