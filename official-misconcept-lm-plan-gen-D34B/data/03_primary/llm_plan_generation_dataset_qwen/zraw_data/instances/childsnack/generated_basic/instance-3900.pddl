; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 585726

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child180 - child
    bread185 - bread-portion
    content284 - content-portion
    tray117 tray393 - tray
    table34 table2 table183 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray117 kitchen)
     (at tray393 kitchen)
     (at_kitchen_bread bread185)
     (at_kitchen_content content284)
     (not_allergic_gluten child180)
     (waiting child180 table34)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child180)
    )
  )
)
