; child-snack task with 1 children and 0.63 gluten factor 
; constant factor of 1.5
; random seed: 218031

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child173 - child
    bread220 - bread-portion
    content201 - content-portion
    tray145 - tray
    table204 table341 table434 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray145 kitchen)
     (at_kitchen_bread bread220)
     (at_kitchen_content content201)
     (not_allergic_gluten child173)
     (waiting child173 table434)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child173)
    )
  )
)
