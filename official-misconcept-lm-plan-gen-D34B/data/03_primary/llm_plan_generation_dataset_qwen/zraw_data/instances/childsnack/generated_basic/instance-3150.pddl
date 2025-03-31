; child-snack task with 2 children and 0.48 gluten factor 
; constant factor of 1.9
; random seed: 670988

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child330 child372 - child
    bread29 bread462 - bread-portion
    content249 content464 - content-portion
    tray363 tray289 - tray
    table213 table227 table485 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray363 kitchen)
     (at tray289 kitchen)
     (at_kitchen_bread bread29)
     (at_kitchen_bread bread462)
     (at_kitchen_content content249)
     (at_kitchen_content content464)
     (not_allergic_gluten child330)
     (not_allergic_gluten child372)
     (waiting child330 table213)
     (waiting child372 table485)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child330)
     (served child372)
    )
  )
)
