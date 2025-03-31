; child-snack task with 1 children and 0.62 gluten factor 
; constant factor of 1.5
; random seed: 260514

(define (problem prob-snack)
  (:domain child-snack)
  (:objects
    child489 - child
    bread287 - bread-portion
    content437 - content-portion
    tray303 tray59 tray110 tray201 - tray
    table471 table103 table280 - place
    sandw1 sandw2 - sandwich
  )
  (:init
     (at tray303 kitchen)
     (at tray59 kitchen)
     (at tray110 kitchen)
     (at tray201 kitchen)
     (at_kitchen_bread bread287)
     (at_kitchen_content content437)
     (not_allergic_gluten child489)
     (waiting child489 table471)
     (notexist sandw1)
     (notexist sandw2)
  )
  (:goal
    (and
     (served child489)
    )
  )
)
