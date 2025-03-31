; child-snack task with 1 children and 0.69 gluten factor 
; constant factor of 1.3
; random seed: 140790

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child198 - child
    bread419 - bread-portion
    content452 - content-portion
    tray498 tray444 tray113 tray395 - tray
    table83 table90 table247 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray498 kitchen)
     (at tray444 kitchen)
     (at tray113 kitchen)
     (at tray395 kitchen)
     (at_kitchen_bread bread419)
     (at_kitchen_content content452)
     (not_allergic_gluten child198)
     (waiting child198 table90)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child198)
    )
  )
)
