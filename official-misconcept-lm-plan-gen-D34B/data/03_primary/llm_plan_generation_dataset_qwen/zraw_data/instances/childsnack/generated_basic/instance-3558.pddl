; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 223248

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child410 child300 - child
    bread449 bread52 - bread-portion
    content259 content267 - content-portion
    tray93 tray129 - tray
    table147 table461 table292 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray93 kitchen)
     (at tray129 kitchen)
     (at_kitchen_bread bread449)
     (at_kitchen_bread bread52)
     (at_kitchen_content content259)
     (at_kitchen_content content267)
     (not_allergic_gluten child410)
     (not_allergic_gluten child300)
     (waiting child410 table461)
     (waiting child300 table147)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child410)
     (served child300)
    )
  )
)
