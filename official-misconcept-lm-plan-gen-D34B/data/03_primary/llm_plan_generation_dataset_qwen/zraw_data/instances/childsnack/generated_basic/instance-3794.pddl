; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 63961

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child255 - child
    bread190 - bread-portion
    content57 - content-portion
    tray334 - tray
    table193 table443 table202 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray334 kitchen)
     (at_kitchen_bread bread190)
     (at_kitchen_content content57)
     (not_allergic_gluten child255)
     (waiting child255 table443)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child255)
    )
  )
)
