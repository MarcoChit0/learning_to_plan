; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 902378

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child267 - child
    bread477 - bread-portion
    content100 - content-portion
    tray373 - tray
    table380 table219 table325 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray373 kitchen)
     (at_kitchen_bread bread477)
     (at_kitchen_content content100)
     (not_allergic_gluten child267)
     (waiting child267 table380)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child267)
    )
  )
)
