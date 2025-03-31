; child-snack task with 1 children and 0.79 gluten factor 
; constant factor of 1.9
; random seed: 456755

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child321 - child
    bread216 - bread-portion
    content496 - content-portion
    tray341 tray374 tray305 - tray
    table313 table403 table234 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray341 kitchen)
     (at tray374 kitchen)
     (at tray305 kitchen)
     (at_kitchen_bread bread216)
     (at_kitchen_content content496)
     (not_allergic_gluten child321)
     (waiting child321 table313)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child321)
    )
  )
)
