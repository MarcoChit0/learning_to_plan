; child-snack task with 1 children and 0.36 gluten factor 
; constant factor of 1.5
; random seed: 918538

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child465 - child
    bread338 - bread-portion
    content283 - content-portion
    tray175 - tray
    table184 table41 table343 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray175 kitchen)
     (at_kitchen_bread bread338)
     (at_kitchen_content content283)
     (not_allergic_gluten child465)
     (waiting child465 table184)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child465)
    )
  )
)
