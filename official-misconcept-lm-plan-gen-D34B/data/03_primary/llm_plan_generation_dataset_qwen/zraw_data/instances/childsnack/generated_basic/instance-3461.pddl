; child-snack task with 2 children and 0.63 gluten factor 
; constant factor of 1.7
; random seed: 469466

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child447 child397 - child
    bread230 bread33 - bread-portion
    content144 content167 - content-portion
    tray466 tray321 - tray
    table394 table206 table128 - place
    sandw1 sandw2 sandw3 sandw4 - sandwich
  )
  (:init
     (at tray466 kitchen)
     (at tray321 kitchen)
     (at_kitchen_bread bread230)
     (at_kitchen_bread bread33)
     (at_kitchen_content content144)
     (at_kitchen_content content167)
     (no_gluten_bread bread33)
     (no_gluten_content content144)
     (allergic_gluten child397)
     (not_allergic_gluten child447)
     (waiting child447 table128)
     (waiting child397 table394)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
     (notexist sandw4)
  )
  (:goal
    (and
     (served child447)
     (served child397)
    )
  )
)
