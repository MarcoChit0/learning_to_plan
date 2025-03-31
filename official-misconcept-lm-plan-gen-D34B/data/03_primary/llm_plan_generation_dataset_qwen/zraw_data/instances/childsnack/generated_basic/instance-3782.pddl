; child-snack task with 1 children and 0.46 gluten factor 
; constant factor of 1.9
; random seed: 6166

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child389 - child
    bread61 - bread-portion
    content19 - content-portion
    tray121 - tray
    table58 table93 table348 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray121 kitchen)
     (at_kitchen_bread bread61)
     (at_kitchen_content content19)
     (not_allergic_gluten child389)
     (waiting child389 table348)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child389)
    )
  )
)
