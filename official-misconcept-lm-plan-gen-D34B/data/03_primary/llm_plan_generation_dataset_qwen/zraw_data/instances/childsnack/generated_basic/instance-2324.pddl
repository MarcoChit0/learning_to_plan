; child-snack task with 1 children and 0.7 gluten factor 
; constant factor of 1.3
; random seed: 18552

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child210 - child
    bread182 - bread-portion
    content223 - content-portion
    tray157 - tray
    table53 table260 table36 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray157 kitchen)
     (at_kitchen_bread bread182)
     (at_kitchen_content content223)
     (not_allergic_gluten child210)
     (waiting child210 table260)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child210)
    )
  )
)
