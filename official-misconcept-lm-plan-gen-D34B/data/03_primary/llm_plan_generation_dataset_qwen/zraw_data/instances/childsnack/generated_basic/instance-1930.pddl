; child-snack task with 1 children and 0.65 gluten factor 
; constant factor of 1.7
; random seed: 55288

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child436 - child
    bread140 - bread-portion
    content435 - content-portion
    tray89 tray239 tray414 - tray
    table1 table484 table459 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray89 kitchen)
     (at tray239 kitchen)
     (at tray414 kitchen)
     (at_kitchen_bread bread140)
     (at_kitchen_content content435)
     (not_allergic_gluten child436)
     (waiting child436 table459)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child436)
    )
  )
)
