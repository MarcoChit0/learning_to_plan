; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 145664

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child443 - child
    bread376 - bread-portion
    content1 - content-portion
    tray447 - tray
    table231 table271 table338 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray447 kitchen)
     (at_kitchen_bread bread376)
     (at_kitchen_content content1)
     (not_allergic_gluten child443)
     (waiting child443 table338)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child443)
    )
  )
)
