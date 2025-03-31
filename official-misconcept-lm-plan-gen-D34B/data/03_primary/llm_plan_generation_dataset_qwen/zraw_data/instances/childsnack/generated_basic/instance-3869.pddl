; child-snack task with 1 children and 0.4 gluten factor 
; constant factor of 1.5
; random seed: 620295

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 - child
    bread240 - bread-portion
    content54 - content-portion
    tray318 tray491 - tray
    table264 table211 table69 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray318 kitchen)
     (at tray491 kitchen)
     (at_kitchen_bread bread240)
     (at_kitchen_content content54)
     (not_allergic_gluten child389)
     (waiting child389 table264)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child389)
    )
  )
)
