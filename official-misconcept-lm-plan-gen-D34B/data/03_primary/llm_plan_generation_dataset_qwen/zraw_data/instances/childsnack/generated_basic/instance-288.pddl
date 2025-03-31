; child-snack task with 1 children and 0.75 gluten factor 
; constant factor of 1.7
; random seed: 488403

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child334 - child
    bread472 - bread-portion
    content221 - content-portion
    tray361 - tray
    table260 table116 table243 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray361 kitchen)
     (at_kitchen_bread bread472)
     (at_kitchen_content content221)
     (not_allergic_gluten child334)
     (waiting child334 table116)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child334)
    )
  )
)
