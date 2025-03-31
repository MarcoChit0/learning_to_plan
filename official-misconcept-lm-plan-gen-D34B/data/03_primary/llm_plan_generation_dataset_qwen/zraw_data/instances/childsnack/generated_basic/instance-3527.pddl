; child-snack task with 2 children and 0.42 gluten factor 
; constant factor of 1.3
; random seed: 689370

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child62 child346 - child
    bread181 bread56 - bread-portion
    content277 content259 - content-portion
    tray344 tray154 - tray
    table440 table158 table274 - place
    sandw1 sandw2 sandw3 - sandwich
  )
  (:init
     (at tray344 kitchen)
     (at tray154 kitchen)
     (at_kitchen_bread bread181)
     (at_kitchen_bread bread56)
     (at_kitchen_content content277)
     (at_kitchen_content content259)
     (not_allergic_gluten child62)
     (not_allergic_gluten child346)
     (waiting child62 table440)
     (waiting child346 table158)
     (notexist sandw1)
     (notexist sandw2)
     (notexist sandw3)
  )
  (:goal
    (and
     (served child62)
     (served child346)
    )
  )
)
