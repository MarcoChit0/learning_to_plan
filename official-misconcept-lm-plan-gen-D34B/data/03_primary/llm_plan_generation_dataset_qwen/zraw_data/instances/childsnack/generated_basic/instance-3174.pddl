; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 751907

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child399 child305 - child
    bread254 bread461 - bread-portion
    content212 content170 - content-portion
    tray220 tray67 - tray
    table144 table472 table111 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray220 kitchen)
     (at tray67 kitchen)
     (at_kitchen_bread bread254)
     (at_kitchen_bread bread461)
     (at_kitchen_content content212)
     (at_kitchen_content content170)
     (not_allergic_gluten child305)
     (not_allergic_gluten child399)
     (waiting child399 table144)
     (waiting child305 table111)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child399)
     (served child305)
    )
  )
)
